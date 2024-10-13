"use client"

import { BusinessExplore } from "@/app/modules/business/BusinessExplore";
import { Business } from "@/types/BusinessTypes";
import { useState, useEffect } from "react";
import { database } from "@../../../firebaseConfig";
import { ref, get, child } from "firebase/database";


export default function forYouPage() {
    const [businesses, setBusinesses] = useState<Business[]>([]);

    const useReadUnseenBusinesses = async () => {
        const dbRef = ref(database);
        const snapshot = await get(child(dbRef, 'unseen'));
        if (snapshot.exists()) {
            return snapshot.val();
        } else {
            console.log("No rejected businesses available");
            return [];
        }
    };

    useEffect(() => {
        const loadBusinesses = async () => {
        const daBusinesses: Business[] = await useReadUnseenBusinesses();
        setBusinesses(daBusinesses);
        }
        loadBusinesses();
    }, []);

    return (
        <div className="bg-[#f4f4f4]">
            <div className="flex justify-center items-center min-h-screen">
                <BusinessExplore businesses={businesses} />
            </div>
        </div>
    );
}