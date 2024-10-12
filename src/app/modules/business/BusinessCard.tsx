import React from "react";

interface BusinessCardProps {
    business: {
        name: string;
        description: string;
        [key: string]: any;
    };
}

/** The card displayed on screen! */
export const BusinessCard: React.FC<BusinessCardProps> = ({ business }) => {


    return (
        <div className="py-1 px-1">
                <h2 className="text-2xl font-bold mb-4">{business.name}</h2>
                <p className="text-gray-700 mb-6">{business.description}</p>
        </div>
    );
};