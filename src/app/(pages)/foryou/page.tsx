"use client"

import { BusinessExplore } from "@/app/modules/business/BusinessExplore";
import Toolbar from "@/app/modules/Toolbar";
import { Business } from "@/types/BusinessTypes";
import { Metadata } from "next";
import { useState, useEffect } from "react";


export default function forYouPage() {
    const [businesses, setBusinesses] = useState<Business[]>([]);

    useEffect(() => {
        const fetchBusinesses = async () => {
            try {
                const response = await fetch('/liked.json');
                const data = await response.json();
                setBusinesses(data);
            } catch (error) {
                console.error('Error fetching businesses:', error);
            }
        };
        fetchBusinesses();
    }, []);

    return (
        <div className="bg-[#f4f4f4]">
            <div className="flex justify-center items-center min-h-screen">
                <BusinessExplore businesses={businesses} />
            </div>
        </div>
    );
}